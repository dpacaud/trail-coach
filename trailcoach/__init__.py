"""trailcoach : lecture Strava et Garmin FIT pour la preparation trail."""
from .config import Athlete, StravaConfig, data_dir, load_dotenv

__all__ = ["Athlete", "StravaConfig", "data_dir", "load_dotenv"]
__version__ = "0.1.0"
