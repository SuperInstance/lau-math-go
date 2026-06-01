FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o lau-math-server .

FROM alpine:3.19
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/lau-math-server /usr/local/bin/

EXPOSE 8080 9090

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost:8080/health || exit 1

ENTRYPOINT ["lau-math-server"]
CMD ["-port", "8080", "-grpc-port", "9090", "-profile", "cloud"]
