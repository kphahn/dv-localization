Object Segmentation
Ground Removal
Euclidean Clustering
Voxel filter - Method of downsampling point clouds


Cone Detection
    1. pre-processing
        Field of View trimming
        data correction (possible distortion)
        ground removal

    2. Cone Detection
        Clustering
        Reconstruction
        Cone legality verification

    3. (Colour Estimation)
        intensity analysis
        camera vision analysis
        
# Literature

feature-based map matching (vs. gps based)
Landmarken-basierte Lokalisierung (Cones)

1. Assozieierung
2. Registrierung (Korrespondenzbasierte Basisregistrierung, ...) https://en.wikipedia.org/wiki/Point-set_registration
3. Robustheit
4. SLAM

Challenges:
    choosing a reasonable epsilon for the dbscan method proves difficult, as the distance to noise is unknown.
    There is, however, an expected number of points per cone - depending on distance an lidar setup

### Registration mechanisms:
- Frontend (icp registration / pose registration)
- Backend (Optimization)
cyrill stachniss https://www.youtube.com/watch?v=uHbRKvD8TWg&t=760s

#### Discussion
Discuss max correspondence distance and effects on registraion / mapping
Graph errors due to measurement errors:
- center of cone actually edge of cone (different centers from different perspectives)
- noise
- frame rate
- uncertainty (information matrix) afffected by all and more factors
- possible information matrices:
    - center line easily determinable, distance travelled on long straights not so much (different uncertainties depending on conditions)
- discussion of optimization methods: LevenbergMarquardt vs. Gauss-Newton
- memory and resource implications (full pose graph requires too much memory)
- how many cones should be taken into account per frame? -> resources, accuracy -> the more cones, the more accurate the icp registration
- correspondence set (icp registration) -> error
- pose graph typically used on smaller maps like rooms

- problems when using real data: spread of points to large. Either single lines get registered as individual cone or multiple cones get identified as one. Too few cones properly detectable to conduct icp registration. sophisticated cone detection required

Initial guess is required for icp registration. However, since start/finish line is distinct, initial guess is relatively simple to implement

### Association mechanisms:
- teaser++
- clipper
