import os, redis
from app.settings import settings

bus = redis.from_url(settings.ORION_BUS_URL)
