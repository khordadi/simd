gcloud compute instances create arm --project=truejit --zone=us-central1-b --machine-type=c4a-standard-4 --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default --maintenance-policy=MIGRATE --provisioning-model=STANDARD --no-service-account --no-scopes --min-cpu-platform=Google\ Axion --create-disk=auto-delete=yes,boot=yes,device-name=arm,image=projects/ubuntu-os-cloud/global/images/ubuntu-2404-noble-arm64-v20260316,mode=rw,provisioned-iops=3060,provisioned-throughput=155,size=10,type=hyperdisk-balanced --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --labels=goog-ec-src=vm_add-gcloud --reservation-affinity=any --performance-monitoring-unit=standard

sudo locale-gen en_GB.UTF-8
sudo update-locale LANG=en_GB.UTF-8

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
    make \
    zsh \
    python3-venv python3-pip \
    ripgrep bat fd-find ncdu htop \
    ranger caca-utils highlight atool w3m poppler-utils mediainfo


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


sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM:=~/.oh-my-zsh/custom}/plugins/zsh-completions

tee ~/.zshrc <<EOF
export PATH=$HOME/bin:/usr/local/bin:$PATH
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="powerlevel10k/powerlevel10k"
plugins=(git docker docker-compose zsh-syntax-highlighting zsh-autosuggestions)
fpath+=${ZSH_CUSTOM:-${ZSH:-~/.oh-my-zsh}/custom}/plugins/zsh-completions/src
export LC_ALL=en_GB.UTF-8
export LANG=en_GB.UTF-8
export LANGUAGE=en_GB.UTF-8
export BAT_PAGER="less -RF"
export EDITOR=nano
alias htop="sudo htop"
alias bat="batcat"
alias fd="fdfind"
alias ncdu="ncdu --color=dark"
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source $HOME/.oh-my-zsh/oh-my-zsh.sh
EOF

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# setup ranger
cd ~
ranger
ranger --copy-config=all

# tmux
cd ~
git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .

# install nano
sudo apt remove -y nano
export VERSION=8.2
export MAJOR_VERSION=$(echo $VERSION | cut -d. -f1)
cd /tmp
wget -O nano.tar.gz https://www.nano-editor.org/dist/v"${MAJOR_VERSION}"/nano-"${VERSION}".tar.gz
tar vfx nano.tar.gz
rm nano.tar.gz
mv nano* nano
cd nano
./configure
make -j$(nproc -a)
sudo make install
curl https://raw.githubusercontent.com/scopatz/nanorc/master/install.sh | sh
cd ..
sudo rm -rf nano


git config --global user.name "Amir Khordadi"
git config --global user.email "amir.khordadi@ed.ac.uk"
