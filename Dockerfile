FROM ic-registry.epfl.ch/mlo/base:ubuntu20.04-cuda110-cudnn8
LABEL omar younis <omar.younis@epfl.ch>


# install some necessary tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        software-properties-common
RUN apt-get install -y \
        inkscape \
        texlive-latex-extra \
        dvipng \
        texlive-full \
        jed \
        libsm6 \
        libxext-dev \
        libxrender1 \
        lmodern \
        libcurl3-dev \
        libfreetype6-dev \
        libzmq3-dev \
        libcupti-dev \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
        locales
RUN apt-get install -y \
        rsync \
        cmake \
        g++ \
        swig \
        vim \
        git \
        curl \
        wget \
        unzip \
        zsh \
        git \
        screen \
        tmux
RUN apt-get install -y openssh-server
# install good vim.
RUN curl http://j.mp/spf13-vim3 -L -o - | sh

# configure environments.
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# configure user.
ENV SHELL=/bin/bash \
    NB_USER=oyounis \
    NB_UID=252255 \
    NB_GROUP=MLO-unit \
    NB_GID=30034
ENV HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}

# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

###### switch to user and compile test example.
USER ${NB_USER}
WORKDIR /home/${NB_USER}


###### switch to root
# expose port for ssh and start ssh service.
EXPOSE 22
# expose port for notebook.
EXPOSE 8888
# expose port for tensorboard.
EXPOSE 6666

RUN sudo mkdir /var/run/sshd
RUN echo 'root:root' | sudo chpasswd
RUN sudo sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sudo sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> sudo /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]


# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
RUN sudo conda create --name=myenv


SHELL ["sudo", "conda", "run", "-n", "myenv", "/bin/bash", "-c"]

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN pip install wandb
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install plotly
RUN pip install pandas

COPY main.py .
COPY async_sgd.py .
COPY train.py .
COPY powersgd/. powersgd/.

# ENTRYPOINT ["sudo", "conda", "run", "--no-capture-output", "-n", "myenv", "python", "main.py"]