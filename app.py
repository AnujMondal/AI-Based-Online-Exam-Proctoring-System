from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import cv2
import numpy as np
import face_recognition
import uuid
import datetime
import threading
import json
import time
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Import our custom modules
from identity_verification import verify_face_match, save_identity_data, get_face_embedding
from audio_monitoring import analyze_audio, save_voice_baseline, detect_multiple_speakers
from object_detection import detect_prohibited_items, toggle_simulate_phone, toggle_simulate_book, simulate_phone, simulate_book
from face_detection import detect_faces_and_landmarks
from head_pose_detection import detect_head_pose, HeadPoseDetector
from session_manager import TestSession, create_session, get_session, end_session, update_session

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['MAX_WARNINGS'] = 5  # Set warning limit to 5 (was 9999)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB max upload size
app.json.encoder = NumpyEncoder  # Use custom JSON encoder

# Increase Werkzeug's max content length for form processing
import werkzeug.formparser
werkzeug.formparser.default_max_content_length = 32 * 1024 * 1024  # 32 MB

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'baseline'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'violations'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'temp'), exist_ok=True)

# Store active test sessions
active_sessions = {}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        password = request.form.get('password')
        
        # In a real app, validate credentials against database
        # For demo, we'll use a simple check
        if student_id and password:
            session['user_id'] = student_id
            print(f"User {student_id} logged in.")
            # Clear any previous session ID before setup
            session.pop('session_id', None)
            return redirect(url_for('setup'))
    
    return render_template('login.html')

@app.route('/setup')
def setup():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if already set up
    if 'session_id' in session and get_session(session['session_id']):
         print(f"User {session['user_id']} already has active session {session['session_id']}, redirecting to test.")
         return redirect(url_for('start_test'))

    print(f"User {session['user_id']} proceeding to setup.")
    return render_template('setup.html')

@app.route('/capture-baseline', methods=['POST'])
def capture_baseline():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'User not logged in'}), 401
    
    user_id = session['user_id']
    print(f"Capturing baseline for user: {user_id}")
    
    try:
        # Process the photo baseline
        photo_data_url = request.form.get('photo_data')
        if not photo_data_url:
            print("Error: No photo data received.")
            return jsonify({'success': False, 'error': 'Photo data is required'}), 400
            
        try:
            # Decode Base64 photo data
            header, encoded = photo_data_url.split(",", 1)
            img_data = base64.b64decode(encoded)
            
            # Save the photo
            baseline_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'baseline')
            photo_filename = f"{user_id}_photo.jpg"
            photo_path = os.path.join(baseline_dir, photo_filename)
            with open(photo_path, 'wb') as f:
                f.write(img_data)
            print(f"Saved baseline photo to: {photo_path}")
            
            # Get face embedding and save
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("Error: Could not decode received image data.")
                return jsonify({'success': False, 'error': 'Invalid image data received'}), 400
                
            embedding = get_face_embedding(img)
            if embedding is None:
                print("Error: No face detected in the baseline photo.")
                # Clean up saved photo if embedding fails
                if os.path.exists(photo_path): os.remove(photo_path)
                return jsonify({'success': False, 'error': 'No face detected in the baseline photo. Please try again.'}), 400
            
            # Save the embedding
            embedding_filename = f"{user_id}_embedding.json"
            embedding_path = os.path.join(baseline_dir, embedding_filename)
            with open(embedding_path, 'w') as f:
                json.dump(embedding.tolist(), f)
            print(f"Saved face embedding to: {embedding_path}")
        except Exception as photo_error:
            print(f"Error processing photo baseline: {photo_error}")
            return jsonify({'success': False, 'error': f'Error processing photo: {photo_error}'}), 500
        
        # Process the voice baseline
        audio_data = request.files.get('audio_data')
        audio_path = None # Default
        if audio_data and audio_data.filename != '':
            try:
                audio_filename = f"{user_id}_voice.wav"
                audio_path = os.path.join(baseline_dir, audio_filename)
                audio_data.save(audio_path)
                print(f"Saved baseline audio to: {audio_path}")
                
                # Create voice fingerprint and save
                save_voice_baseline(audio_path, user_id)
            except Exception as audio_error:
                print(f"Warning: Could not process voice baseline: {audio_error}")
                # Fallback: Still create session, but audio checks might be skipped later
                save_voice_baseline(None, user_id) # Indicate no valid audio baseline
        else:
            print("No voice baseline provided or bypassed by user.")
            save_voice_baseline(None, user_id) # Ensure function is called to handle potential state

        # Create test session
        session_id = create_session(user_id)
        session['session_id'] = session_id
        print(f"Created new test session {session_id} for user {user_id}")
        
        return jsonify({'success': True, 'session_id': session_id})
        
    except Exception as e:
        # Catch-all for unexpected errors during baseline capture
        print(f"Unexpected error during capture_baseline for user {user_id}: {e}")
        return jsonify({'success': False, 'error': f'An unexpected server error occurred: {e}'}), 500

@app.route('/start-test')
def start_test():
    if 'user_id' not in session:
        print("User not logged in, redirecting to login.")
        return redirect(url_for('login'))
    if 'session_id' not in session:
         print(f"User {session['user_id']} has no session_id, redirecting to setup.")
         return redirect(url_for('setup')) # Redirect to setup if baseline wasn't completed

    # Make sure simulations are disabled when starting a test
    from object_detection import simulate_phone, simulate_book
    if simulate_phone or simulate_book:
        print(f"Disabling simulations for test session: phone={simulate_phone}, book={simulate_book}")
        if simulate_phone:
            toggle_simulate_phone()
        if simulate_book:
            toggle_simulate_book()

    print(f"User {session['user_id']} starting test with session {session['session_id']}.")
    return render_template('test.html', session_id=session['session_id'])

@app.route('/monitor', methods=['POST'])
def monitor():
    start_time = time.time()
    # Log monitoring request for debugging
    print(f"Received monitoring request: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Extract data
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    session_id = data.get('session_id')
    user_id = data.get('user_id')
    exam_id = data.get('exam_id')
    frame_data = data.get('frame')
    audio_data = data.get('audio')
    
    if not session_id or not frame_data:
        return jsonify({'error': 'Missing required data'}), 400
    
    # Decode base64 frame
    try:
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return jsonify({'error': 'Invalid frame data'}), 400
    
    # Initialize results dictionary
    results = {
        'face_detected': False,
        'face_match': False,
        'multiple_speakers': False,
        'prohibited_objects': [],
        'head_pose': 'Not Available',
        'similarity_score': 0,
        'terminate': False,
        'warning_message': ''
    }
    
    # Process audio if provided
    if audio_data:
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            
            # For debugging only: Save audio to file
            # with open(f"audio_sample_{time.time()}.wav", "wb") as f:
            #     f.write(audio_bytes)
            
            # Detect multiple speakers in audio
            results['multiple_speakers'] = audio_monitoring.detect_multiple_speakers(audio_bytes, user_id)
        except Exception as e:
            print(f"Error processing audio: {e}")
    
    # Override multiple_speakers for demo - fully disable this feature
    results['multiple_speakers'] = False
    
    # Detect faces in the frame
    faces = face_detection.detect_faces(frame)
    
    if faces:
        results['face_detected'] = True
        
        # Only verify identity if user_id is provided
        if user_id:
            # Verify face against stored reference
            match, similarity = identity_verification.verify_face_match(frame, faces[0], user_id)
            results['face_match'] = match
            results['similarity_score'] = similarity
        
        # Detect head pose
        head_detector = HeadPoseDetector()
        is_normal, angle, message = head_detector.detect_head_pose(frame)
        results['head_pose'] = message
        
        # Less sensitive head pose detection - only trigger warning for significant deviations
        if not is_normal and abs(angle) > 25:  # Increased threshold from default
            results['warning_message'] = f"Warning: Head pose issue: {message} (angle: {angle})"
        
        # Detect objects
        results['prohibited_objects'] = object_detection.detect_prohibited_items(frame)
    
    # Prepare warning message based on results
    warning_parts = []
    
    if not results['face_detected']:
        warning_parts.append("Face not detected")
    elif not results['face_match']:
        warning_parts.append("Identity verification failed")
    
    if results['multiple_speakers']:
        warning_parts.append("Multiple voices detected")
    
    if results['prohibited_objects']:
        objects_str = ", ".join(results['prohibited_objects'])
        warning_parts.append(f"Prohibited items detected: {objects_str}")
    
    if results['head_pose'] != 'Normal' and results['head_pose'] != 'Not Available' and results['head_pose'] != 'No Face':
        # Customize warning message based on eye gaze/head pose
        if 'Looking Left' in results['head_pose'] or 'Looking Right' in results['head_pose']:
            warning_parts.append("Please look straight at your screen")
        elif 'Looking Down' in results['head_pose']:
            warning_parts.append("Please look up at your screen, not down")
        elif 'Looking Up' in results['head_pose']:
            warning_parts.append("Please look directly at your screen")
        elif 'Tilted' in results['head_pose']:
            warning_parts.append("Please keep your head upright")
        else:
            warning_parts.append(f"Incorrect head position: {results['head_pose']}")
    
    # Combine all warning parts
    if warning_parts:
        results['warning_message'] = ". ".join(warning_parts)
    
    # Determine if the exam should be terminated
    # Currently we only terminate for prohibited objects or consistent identity verification failure
    # This should be determined by the client-side rules based on warning count
    if len(results['prohibited_objects']) > 0:
        results['terminate'] = False  # For demo, we'll just warn
    
    # Periodic screenshot for audit trail (every 30 seconds or based on suspicious activity)
    timestamp = int(time.time())
    save_screenshot = (timestamp % 30 == 0) or results['terminate'] or len(warning_parts) > 0
    
    if save_screenshot:
        try:
            session_dir = os.path.join('static', 'uploads', 'sessions', session_id)
            if not os.path.exists(session_dir):
                os.makedirs(session_dir, exist_ok=True)
            
            filename = f"{session_dir}/screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            
            # Log saved screenshot
            print(f"Saved screenshot to {filename} for session {session_id}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
    
    # Log results for debugging
    print(f"Monitoring results: {results}")
    
    # Log processing time
    elapsed = time.time() - start_time
    print(f"Processing time: {elapsed:.2f} seconds")
    
    return jsonify(results)

@app.route('/api/end-test', methods=['POST'])
def end_test():
    if 'session_id' not in session:
        print("End-test request failed: No session_id in session.")
        return jsonify({'success': False, 'error': 'No active session'}), 401
    
    session_id = session['session_id']
    user_id = session.get('user_id', 'Unknown')
    print(f"Ending test session {session_id} for user {user_id} via API.")
    
    end_session(session_id, "User completed test")
    
    # Clean up session variables after ending test
    session.pop('session_id', None)
    # Keep user_id for potential logout message or redirect
    # session.pop('user_id', None) 
    
    return jsonify({'success': True})

@app.route('/logout')
def logout():
    user_id = session.get('user_id', 'Unknown')
    print(f"Logging out user {user_id}.")
    session.clear() # Clear all session data
    return redirect(url_for('login'))

@app.route('/simple-test')
def simple_test():
    return render_template('simple_test.html')

@app.route('/api/toggle-simulate-phone', methods=['POST'])
def api_toggle_simulate_phone():
    """API endpoint to toggle phone simulation"""
    is_enabled = toggle_simulate_phone()
    return jsonify({'success': True, 'phone_simulation': is_enabled})

@app.route('/api/toggle-simulate-book', methods=['POST'])
def api_toggle_simulate_book():
    """API endpoint to toggle book simulation"""
    is_enabled = toggle_simulate_book()
    return jsonify({'success': True, 'book_simulation': is_enabled})

@app.route('/api/disable-all-simulations', methods=['GET', 'POST'])
def disable_all_simulations():
    """Disable all simulation toggles"""
    global simulate_phone, simulate_book
    from object_detection import simulate_phone, simulate_book
    
    # Explicitly reset simulation flags
    if simulate_phone:
        toggle_simulate_phone()
    if simulate_book:
        toggle_simulate_book()
        
    return jsonify({
        'success': True, 
        'phone_simulation': simulate_phone,
        'book_simulation': simulate_book,
        'message': 'All simulations disabled'
    })

@app.route('/api/simulation-status', methods=['GET'])
def simulation_status():
    """Get the current status of simulation toggles"""
    from object_detection import simulate_phone, simulate_book
    
    return jsonify({
        'success': True,
        'phone_simulation': simulate_phone,
        'book_simulation': simulate_book
    })

@app.route('/api/monitor', methods=['POST', 'GET'])
def api_monitor():
    start_time = time.time()
    # Log monitoring request for debugging
    print(f"Received API monitoring request: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Handle GET requests (easier to troubleshoot than multipart form POST)
        if request.method == 'GET':
            session_id = request.args.get('session_id')
            
            # For GET requests, just return simulated results
            return jsonify({
                'success': True,
                'results': {
                    'face_count': 1,
                    'identity_match': True,
                    'head_pose': 'Normal head pose',
                    'gaze_direction': 'Straight',
                    'multiple_speakers': False,
                    'different_speaker': False,
                    'voice_match_confidence': 0.9,
                    'prohibited_items': []
                },
                'warnings': 0,
                'terminate': False,
                'reason': '',
                'processing_time': 0.01
            })
        
        # First try to get session_id from query params (safer than form parsing)
        session_id = request.args.get('session_id')
        
        # If not in query params, try the form data (which might fail with large requests)
        if not session_id:
            try:
                session_id = request.form.get('session_id')
            except Exception as e:
                print(f"Error parsing form data: {e}")
                session_id = None
        
        if not session_id:
            print(f"WARNING: Could not determine session_id from query params or form data: {session_id}")
        
        # Get the user_id from the session
        user_id = None
        if 'user_id' in session:
            user_id = session['user_id']
            print(f"Using user_id {user_id} from session cookie")
        elif session_id:
            # Try to get the user_id from the session_id
            from session_manager import get_session
            test_session = get_session(session_id)
            if test_session:
                user_id = test_session.user_id
                print(f"Retrieved user_id {user_id} from session {session_id}")
            else:
                print(f"WARNING: No test session found for session_id: {session_id}")
        
        if not user_id:
            print(f"WARNING: Could not determine user_id from session or session_id: {session_id}")
        
        # Get the frame data from the form
        frame_data = request.form.get('frame_data')
        
        # Get audio data from the form
        audio_data = request.form.get('audio_data')
        
        if not frame_data:
            print("Error: No frame data received in API request")
            return jsonify({
                'success': False, 
                'error': 'No frame data provided'
            }), 400
            
        # Decode the frame data
        try:
            # Extract the base64 part (remove data:image prefix if present)
            if ',' in frame_data:
                _, encoded = frame_data.split(",", 1)
            else:
                encoded = frame_data
                
            # Decode Base64 data
            frame_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Could not decode frame data")
                
        except Exception as e:
            print(f"Error decoding frame in API: {str(e)}")
            return jsonify({
                'success': False, 
                'error': f'Invalid frame data: {str(e)}'
            }), 400
        
        # Initialize results
        monitoring_results = {
            'face_count': 0,
            'identity_match': False,
            'head_pose': 'Unknown',
            'gaze_direction': 'Unknown',
            'multiple_speakers': False,
            'different_speaker': False,
            'voice_match_confidence': 0.0,
            'prohibited_items': []
        }
        
        warnings = 0
        terminate_exam = False
        reason = ""
        
        # Check if we should retrieve previous warning count from session
        if session_id:
            from session_manager import get_session, update_session
            test_session = get_session(session_id)
            if test_session:
                # Get previous warning count from session
                warnings = getattr(test_session, 'warning_count', 0)
                print(f"Retrieved previous warning count: {warnings}")
            
        # Run face detection
        faces = detect_faces_and_landmarks(frame)
        
        if faces:
            monitoring_results['face_count'] = len(faces)
            
            # Identity verification (if reference exists)
            if user_id and len(faces) == 1:
                match, similarity = verify_face_match(frame, None, user_id)
                monitoring_results['identity_match'] = match
                monitoring_results['similarity_score'] = similarity
                
                if not match and similarity < 0.4:  # Stricter threshold for warnings
                    warnings += 1
                    if similarity < 0.2:  # Very low similarity may indicate an impostor
                        terminate_exam = True
                        reason = "Identity verification failed. Possible impersonation detected."
            
            # Detect head pose for the first face
            head_detector = HeadPoseDetector()
            is_normal, angle, message = head_detector.detect_head_pose(frame)
            monitoring_results['head_pose'] = message
            
            # Less sensitive head pose detection - only trigger warning for significant deviations
            if not is_normal and abs(angle) > 25:  # Increased threshold from default
                warnings += 1
                print(f"Warning: Head pose issue: {message} (angle: {angle})")
            
            # Add eye gaze detection
            try:
                is_looking_straight, gaze_direction, confidence, gaze_message = head_detector.detect_eye_gaze(frame)
                monitoring_results['gaze_direction'] = gaze_direction
                monitoring_results['gaze_message'] = gaze_message
                monitoring_results['gaze_confidence'] = float(confidence)
                
                # Reduced sensitivity for gaze detection
                if not is_looking_straight and confidence > 0.8:  # Increased from 0.6
                    warnings += 1
                    print(f"Warning: Eye gaze not straight: {gaze_message} (confidence: {confidence:.2f})")
            except Exception as e:
                print(f"Error in eye gaze detection: {str(e)}")
                monitoring_results['gaze_direction'] = "Straight"
                monitoring_results['gaze_message'] = "Looking straight ahead (error recovery)"
                monitoring_results['gaze_confidence'] = 0.5
                
            # Detect prohibited items
            prohibited_items = detect_prohibited_items(frame)
            
            # Add detailed logging for prohibited items detection
            print(f"Prohibited items detection result: {prohibited_items}")
            print(f"Simulation status - Phone: {simulate_phone}, Book: {simulate_book}")
            
            monitoring_results['prohibited_items'] = prohibited_items
            
            if prohibited_items:
                warnings += 1
                print(f"WARNING: Prohibited items detected: {prohibited_items}")
                if 'phone' in prohibited_items:
                    terminate_exam = True
                    reason = "Prohibited item detected: Phone"
                    print("EXAM TERMINATION TRIGGERED: Phone detected")
        else:
            # Only add face not detected warning after several consecutive failures
            # This helps prevent false positives from momentary camera issues
            print("No face detected in current frame")
            # Start with a face count of 0 but don't immediately add a warning
            monitoring_results['face_count'] = 0
        
        # Process audio if provided
        if audio_data:
            try:
                # Decode base64 audio
                audio_bytes = base64.b64decode(audio_data.split(',')[1])
                
                # Log audio size for debugging
                audio_size_kb = len(audio_bytes) / 1024
                print(f"Received audio data: {audio_size_kb:.2f} KB")
                
                if audio_size_kb > 1024:  # If larger than 1MB
                    print(f"Audio size too large ({audio_size_kb:.2f} KB), skipping processing")
                else:
                    # Analyze audio including voice verification
                    audio_results = analyze_audio(audio_bytes, user_id)
                    
                    # Update results with audio analysis
                    monitoring_results['multiple_speakers'] = audio_results.get('multiple_speakers', False)
                    monitoring_results['different_speaker'] = audio_results.get('different_speaker', False)
                    monitoring_results['voice_match_confidence'] = audio_results.get('voice_match_confidence', 0.0)
                    
                    # Add a warning if voice confidence is very low (below 20%)
                    voice_confidence = audio_results.get('voice_match_confidence', 0.0)
                    if voice_confidence < 0.2:
                        warnings += 1
                        reason = f"Voice authentication failed - low confidence ({int(voice_confidence*100)}%)"
                        print(f"WARNING: Voice confidence too low: {voice_confidence:.2f}")
            except Exception as e:
                print(f"Error processing audio: {e}")
        
        # Voice authentication simulation - more stable with higher success rate
        voice_sim_value = random.random()
        if voice_sim_value > 0.85:  # Only 15% chance of failing verification
            monitoring_results['different_speaker'] = True
            monitoring_results['voice_match_confidence'] = round(random.uniform(0.3, 0.5), 2)
        else:
            monitoring_results['different_speaker'] = False
            monitoring_results['voice_match_confidence'] = round(random.uniform(0.7, 0.95), 2)
        
        # Override multiple_speakers for demo - fully disable this feature
        monitoring_results['multiple_speakers'] = False
        
        # ONLY use simulation if specifically enabled by the user toggle,
        # don't randomly generate false positives
        if simulate_phone or simulate_book:
            simulation_items = []
            if simulate_phone:
                simulation_items.append("phone")
                print("Adding simulated phone to detection results")
            if simulate_book:
                simulation_items.append("book")
                print("Adding simulated book to detection results")
                
            # Only override if we have items to add
            if simulation_items:
                print(f"Setting prohibited_items to simulation items: {simulation_items}")
                monitoring_results['prohibited_items'] = simulation_items
                
                # Make sure to set termination if phone is simulated
                if 'phone' in simulation_items:
                    terminate_exam = True
                    reason = "Prohibited item detected: Phone"
                    print("SIMULATION: EXAM TERMINATION TRIGGERED by simulated phone")
                    
                # Also add warnings
                warnings += 1
        
        # Check if warnings exceed the max limit
        MAX_WARNINGS = app.config['MAX_WARNINGS']
        if warnings >= MAX_WARNINGS and not terminate_exam:
            terminate_exam = True
            reason = f"Maximum warnings ({MAX_WARNINGS}) exceeded"
            print(f"EXAM TERMINATION TRIGGERED: Maximum warnings ({warnings}/{MAX_WARNINGS}) exceeded")
        
        # Update session with new warning count if session exists
        if session_id:
            from session_manager import get_session, update_session
            test_session = get_session(session_id)
            if test_session:
                test_session.warning_count = warnings
                update_session(test_session)
                print(f"Updated session with warning count: {warnings}")
                
        # Log final values
        print(f"Final result - prohibited_items: {monitoring_results['prohibited_items']}, terminate: {terminate_exam}, reason: {reason}, warnings: {warnings}/{MAX_WARNINGS}")

        # Return the results
        response = {
            'success': True,
            'results': monitoring_results,
            'warnings': int(warnings) if not isinstance(warnings, int) else warnings,
            'terminate': bool(terminate_exam) if not isinstance(terminate_exam, bool) else terminate_exam,
            'reason': reason,
            'processing_time': float(round(time.time() - start_time, 3))
        }
        
        # Ensure all values are JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
                
        # Convert the entire response to be JSON serializable
        response = make_json_serializable(response)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Unexpected error in API monitoring: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/verify-voice', methods=['POST'])
def verify_voice():
    """Separate endpoint for voice verification to avoid overburdening monitoring requests"""
    start_time = time.time()
    
    try:
        # Get session ID
        session_id = request.form.get('session_id') or request.args.get('session_id')
        
        # Get audio data
        audio_data = request.form.get('audio_data')
        
        if not audio_data:
            return jsonify({
                'success': False,
                'error': 'No audio data provided'
            }), 400
            
        # Get the user_id from the session
        user_id = None
        if 'user_id' in session:
            user_id = session['user_id']
        elif session_id:
            # Try to get the user_id from the session_id
            from session_manager import get_session
            test_session = get_session(session_id)
            if test_session:
                user_id = test_session.user_id
                
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'Could not determine user ID'
            }), 400
            
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Analyze audio
        audio_results = analyze_audio(audio_bytes, user_id)
        
        # Ensure values are JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        # Create response
        response = {
            'success': True,
            'different_speaker': bool(audio_results.get('different_speaker', False)),
            'voice_match_confidence': float(audio_results.get('voice_match_confidence', 0.0)),
            'processing_time': float(round(time.time() - start_time, 3))
        }
        
        # Return voice verification results
        return jsonify(make_json_serializable(response))
        
    except Exception as e:
        print(f"Error in voice verification endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting Enhanced Proctoring Flask App...")
    
    # Import necessary modules
    import socket
    import random
    
    # More aggressive simulation disable on startup
    from object_detection import simulate_phone, simulate_book, toggle_simulate_phone, toggle_simulate_book
    
    # Force disable regardless of current state
    print("Forcefully disabling all simulations on startup...")
    # Reset the global variables directly
    import object_detection
    object_detection.simulate_phone = False
    object_detection.simulate_book = False
    print(f"Simulation flags reset: phone={object_detection.simulate_phone}, book={object_detection.simulate_book}")
    
    # Check if port 5004 is available, if not try other ports
    port = 5004
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('127.0.0.1', port))
            s.close()
            break
        except socket.error:
            print(f"Port {port} is in use, trying {port+1}")
            port += 1
    
    print(f"Starting Enhanced Proctoring Flask App on port {port}...")
    app.run(debug=True, port=port, threaded=True) 