ARG IMAGE=nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04

FROM ${IMAGE} as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -yq build-essential wget libncurses5-dev libncursesw5-dev libssl-dev \
  pkg-config libdrm-dev libgtest-dev libudev-dev python3-venv git

# Get a recent-enough CMake
RUN python3 -m venv /.venv && \
    . /.venv/bin/activate && \
    pip install --upgrade pip && \
    pip install cmake

RUN git clone https://github.com/Syllo/nvtop.git /nvtop
WORKDIR /nvtop
RUN mkdir -p /nvtop/build && \
  cd /nvtop/build && \
  . /.venv/bin/activate && \
  cmake .. && \
  make -j && \
  make install

FROM python:3.12-slim AS linux-base

COPY --from=builder /usr/local/bin/nvtop /usr/local/bin/nvtop
COPY --from=builder /usr/local/share/man/man1/nvtop.1 /usr/local/share/man/man1/nvtop.1

ARG _USERNAME=driverless
ENV USERNAME=${_USERNAME}

RUN groupadd --gid 1000 $_USERNAME \
    && useradd --uid 1000 --gid 1000 -m $_USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo wget curl git htop less rsync screen vim nano wget build-essential software-properties-common python3-launchpadlib libfuse2 \
    && echo $_USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$_USERNAME \
    && chmod 0440 /etc/sudoers.d/$_USERNAME \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/${_USERNAME}
USER $_USERNAME

# Environment variables
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=python3.12
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"


# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
