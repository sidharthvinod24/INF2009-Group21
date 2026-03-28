# camera package
from .camera_thread import CameraThread
from .pir_thread    import PIRWatchThread, pir_lock, wake_requested, camera_active, last_camera_off_time, camera_last_person_time
