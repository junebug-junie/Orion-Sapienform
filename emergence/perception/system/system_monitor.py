# perception/system/system_monitor.py
import psutil
import uuid
from datetime import datetime
from emergence.memory.interface import write_to_memory
from emergence.core.redis_bus import RedisBus
from emergence.schema.context import MemoryContext
from emergence.cognition.introspection.agent_introspector import AgentIntrospector

class SystemMonitor:
    def __init__(self, name="system-monitor", memory=None):
        self.name = name
        self.memory = memory
        self.bus = RedisBus()
        self.introspector = AgentIntrospector(name=name, memory=memory)

    def check_vitals(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent

        timestamp = datetime.utcnow().isoformat()
        memory_id = str(uuid.uuid4())

        entry = {
            "memory_id": memory_id,
            "observer": self.name,
            "type": "system",
            "emergent_entity": "MachineVitals",
            "summary": f"Vitals: CPU={cpu}%, Mem={mem}%, Disk={disk}%",
            "timestamp": timestamp,
            "context": MemoryContext(agent_id=self.name).to_dict(),
            "cpu": cpu,
            "memory": mem,
            "disk": disk
        }

        write_to_memory(entry)
        self.bus.publish("system:vitals", entry)

        # Trigger introspection if anything is high
        if cpu > 85 or mem > 90 or disk > 95:
            self.introspector.reflect(
                prompt=f"System strain detected: CPU={cpu}%, Mem={mem}%, Disk={disk}%",
                salience=0.7,
                extra=entry
            )

        return entry

