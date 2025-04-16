import uuid
import datetime
import json
import os
import time

# Directory for session logs
SESSION_DIR = 'static/uploads/sessions'
os.makedirs(SESSION_DIR, exist_ok=True)

# Active sessions in memory
active_sessions = {}

class TestSession:
    """
    Class to manage a test session, including tracking violations and warnings
    """
    def __init__(self, session_id, user_id):
        self.session_id = session_id
        self.user_id = user_id
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.is_active = True
        self.warning_count = 0
        self.violation_log = []
        self.termination_reason = None
        
        # Track activity metrics
        self.face_disappearances = 0
        self.face_appearance_rate = 100.0  # Percentage of time face is visible
        self.identity_mismatch_count = 0
        self.prohibited_items_detected = []
        self.head_pose_issues = []
        self.audio_issues = []
        
        # Last activity timestamp
        self.last_activity = time.time()
        
        # Log creation
        self.log_activity("Session started")
    
    def log_activity(self, activity, details=None):
        """Log an activity to the session's violation log"""
        timestamp = datetime.datetime.now()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "activity": activity,
            "details": details
        }
        self.violation_log.append(log_entry)
        
        # Update last activity timestamp
        self.last_activity = time.time()
        
        return log_entry
    
    def log_violation(self, violation_type, details=None):
        """Log a violation and increment warning count"""
        self.warning_count += 1
        
        # Track specific violation types
        if violation_type == "face_disappeared":
            self.face_disappearances += 1
        elif violation_type == "identity_mismatch":
            self.identity_mismatch_count += 1
        elif violation_type == "prohibited_item":
            if details and "item" in details:
                self.prohibited_items_detected.append(details["item"])
        elif violation_type == "head_pose":
            if details and "pose" in details:
                self.head_pose_issues.append(details["pose"])
        elif violation_type == "audio_issue":
            if details and "issue" in details:
                self.audio_issues.append(details["issue"])
        
        # Log the violation
        self.log_activity(f"Violation: {violation_type}", details)
        
        return self.warning_count
    
    def calculate_metrics(self):
        """Calculate session metrics based on logged activities"""
        if not self.is_active and self.end_time and self.start_time:
            session_duration = (self.end_time - self.start_time).total_seconds()
            
            # Calculate additional metrics here if needed
            return {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "duration_seconds": session_duration,
                "warning_count": self.warning_count,
                "face_disappearances": self.face_disappearances,
                "identity_mismatches": self.identity_mismatch_count,
                "prohibited_items": self.prohibited_items_detected,
                "head_pose_issues": self.head_pose_issues,
                "audio_issues": self.audio_issues,
                "terminated_early": self.termination_reason is not None,
                "termination_reason": self.termination_reason
            }
        
        return None
    
    def end_session(self, reason=None):
        """End the session and calculate final metrics"""
        if self.is_active:
            self.is_active = False
            self.end_time = datetime.datetime.now()
            self.termination_reason = reason
            
            # Log session end
            self.log_activity("Session ended", {"reason": reason})
            
            # Calculate final metrics
            metrics = self.calculate_metrics()
            
            # Save session log to file
            self.save_session_log()
            
            return metrics
        
        return None
    
    def save_session_log(self):
        """Save the session log to a file"""
        try:
            log_file = os.path.join(SESSION_DIR, f"{self.session_id}.json")
            
            # Prepare data for serialization
            session_data = {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "is_active": self.is_active,
                "warning_count": self.warning_count,
                "violation_log": self.violation_log,
                "metrics": self.calculate_metrics()
            }
            
            with open(log_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving session log: {e}")
            return False

def create_session(user_id):
    """Create a new test session and return the session ID"""
    session_id = str(uuid.uuid4())
    session = TestSession(session_id, user_id)
    active_sessions[session_id] = session
    
    print(f"Created new session {session_id} for user {user_id}")
    return session_id

def get_session(session_id):
    """Get a session by ID, returns None if not found"""
    return active_sessions.get(session_id)

def end_session(session_id, reason=None):
    """End a session and return metrics"""
    session = get_session(session_id)
    if session:
        metrics = session.end_session(reason)
        
        # Consider removing from active_sessions after some time
        # For now, keep it but marked as inactive
        
        return metrics
    
    return None

def cleanup_old_sessions(max_age_hours=24):
    """Clean up old inactive sessions"""
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, session in active_sessions.items():
        # If session is inactive or hasn't had activity in max_age_hours
        if (not session.is_active or 
            (current_time - session.last_activity) > max_age_hours * 3600):
            sessions_to_remove.append(session_id)
    
    # Remove old sessions
    for session_id in sessions_to_remove:
        # Make sure session log is saved before removing
        active_sessions[session_id].save_session_log()
        del active_sessions[session_id]
    
    return len(sessions_to_remove)

# Start a background thread to periodically clean up old sessions
def start_cleanup_thread():
    import threading
    
    def cleanup_task():
        while True:
            cleanup_old_sessions()
            time.sleep(3600)  # Run every hour
    
    cleanup_thread = threading.Thread(target=cleanup_task)
    cleanup_thread.daemon = True
    cleanup_thread.start()

# Start the cleanup thread when this module is imported
start_cleanup_thread()

def update_session(session):
    """Update an existing session in the active_sessions dictionary"""
    if session and session.session_id in active_sessions:
        active_sessions[session.session_id] = session
        
        # Save the session log to persist changes
        session.save_session_log()
        
        return True
    
    return False 