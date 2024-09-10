
echo "Installing Ambianic dependencies with YOLOv8 support"

set -e

set -x

apt-get update -y && apt-get install -y sudo

if python3 --version
then
  echo "python3 is already installed."
else
  echo "python3 is not available from the parent image. Installing python3 now."
  sudo apt-get install -y python3 && apt-get install -y python3-pip
fi

sudo apt-get install -y python3-numpy
sudo apt-get install -y libjpeg-dev zlib1g-dev
sudo apt-get install -y libgl1-mesa-glx  

python3 -m pip install --upgrade pip
pip3 --version

pip3 install ultralytics
pip3 install opencv-python-headless
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip3 install -r requirements.txt

echo "YOLOv8 and its dependencies have been installed"

pip3 list
sudo apt-get -y autoremove

sudo rm -rf /var/lib/apt/lists/*

sudo apt-get clean

echo "Installation complete"
