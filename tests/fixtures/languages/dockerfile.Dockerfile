FROM rust:1.88-slim AS builder
ARG DEBIAN_FRONTEND=noninteractive
ENV CARGO_HOME=/cargo \
    RUSTFLAGS="-C debuginfo=0"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/betlang /usr/local/bin/betlang
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/betlang"]
CMD ["--help"]
