# this installs the right pip and dependencies for the fresh python
conda install ipython pip -y 

# python dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX

# install PyTorch
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch -y

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# clone repo
cd $INSTALL_DIR
cd box_of_ai_tools/Text_Recognition/MaskTextSpotter

# build
python setup.py build develop

unset INSTALL_DIR
echo "download the pretrained model and put it in the MaskTextSpotter dir, Link to model: https://drive.google.com/file/d/1pPRS7qS_K1keXjSye0kksqhvoyD0SARz/view "
echo "Now put your images in the demo_images folder and press any key to see the demo."
while [ true ] ; do 
  read -t 3 -n 1
if [ $? = 0 ] ; then 
  python tools/demo.py 
