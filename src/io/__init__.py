"""
Input/Output module - minimal version.
Only safe modules are exported.
"""

# Safe imports only
try:
    from .readers import *
except ImportError:
    pass

try:
    from .writers import *
except ImportError:
    pass

try:
    from .validators import *
except ImportError:
    pass

# Database connectors disabled due to syntax error
# from .database_connectors import ...

# Geological formats disabled for now
# from .geological_formats import ...

# Specialized exports disabled for now  
# from .specialized_exports import ...

# Report generators disabled for now
# from .report_generators import ...