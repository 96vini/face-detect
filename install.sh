git clone git@github.com:pytorch/vision.git

conda install -c conda-forge 'ffmpeg<4.3'

python setup.py install

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

mkdir build_opencv
cd build_opencv

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON ../opencv

cd ..

pip install -r requirements.txt