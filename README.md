# visual_odometry_estimator

🤖 🛣️ C++ application implementing pipeline for estimating odometry of robot based on video stream from camera (Visual odometry).

Task: given a set of frame images from monocular camera we need to estimate the 6-DOF trajectory passed by the robot. 

## Using package

Stack:
- OpenCV
- Eigen
- FLANN
- Conan

Install:

```
conan install .
conan build .
```

## Run tests

Move to folder bin and run tests:

```
cd bin && ./library_test
```
