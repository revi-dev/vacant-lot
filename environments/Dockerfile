ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1

# ロケール設定
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

# タイムゾーン設定
ENV TZ=Asia/Tokyo
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# パッケージの更新
RUN apt update && apt upgrade -y
RUN apt install --no-install-recommends -y \
                wget curl git \
                build-essential libbz2-dev libdb-dev libreadline-dev \
                libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev \
                libsqlite3-dev libssl-dev zlib1g-dev uuid-dev \
                libgl1-mesa-dev libglib2.0-0


ARG USER_NAME
ARG GROUP_NAME
ARG UID
ARG GID

# ユーザー追加
RUN groupadd -g ${GID} ${GROUP_NAME} && \
    useradd -ms /bin/bash -u ${UID} -g ${GID} ${USER_NAME}

# 以降, ユーザーでの操作
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/workspace


# uvのインストール
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && echo 'source "$HOME/.cargo/env"' >> ~/.bashrc


CMD ["/bin/bash"]
