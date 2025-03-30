
# VILAS: Visual-Inertial Landmark Adaptive System

VILAS provides high-speed drone navigation in GNSS-denied environments by intelligently combining inertial tracking with selective landmark recognition. The system continuously tracks position using IMU data processed through a Kalman filter while conserving resources by activating its visual navigation component only when flying over pre-identified high-feature areas. When landmarks are detected, the system applies position corrections to reset accumulated drift. If expected landmarks aren't found, VILAS employs SimCLR (Simple Contrastive Learning of Representations) to analyze visual features and determine position based on learned terrain representations feature-rich areas. This multi-tiered approach ensures reliable positioning at speeds exceeding 300 km/h using only off-the-shelf components, balancing accuracy with computational efficiency for resource-constrained platforms operating in challenging environments



