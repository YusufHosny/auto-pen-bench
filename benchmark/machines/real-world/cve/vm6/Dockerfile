FROM openjdk:8u342-jre-slim-buster

RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    sed -i '/stretch-updates/d' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list

RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       libfindbin-libs-perl \
       wget \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/druid \
    && wget -qO- https://archive.apache.org/dist/druid/0.20.0/apache-druid-0.20.0-bin.tar.gz | tar xz --strip-components 1 -C /opt/druid

COPY flag.txt /root/flag.txt

EXPOSE 8888
WORKDIR /opt/druid
CMD [ "/opt/druid/bin/start-nano-quickstart" ]