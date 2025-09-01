wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2004-9.1.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.1.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get -y install cudnn
echo "check path: find /usr -name \"*cudnn*\" 2>/dev/null | grep \"\.so\.9\""
echo "set env var: export LD_LIBRARY_PATH=<PATH TO CUDNN FILES>:$LD_LIBRARY_PATH"