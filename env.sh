git config --global user.name "Amir Khordadi"
git config --global user.email "amir.khordadi@ed.ac.uk"

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    curl \
    unzip \
    git \
    lsb-release \
    wget \
    software-properties-common \
    gnupg \
    make

cd
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 22 all
rm llvm.sh


BINS="clang clang++ clangd clang-format llvm-objdump"
for BIN in $BINS; do
    sudo ln -s /usr/bin/$BIN-22 /usr/bin/$BIN
done

# create .venv
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip cache purge
