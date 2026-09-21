# PowerShell script to launch Qdrant container with persistent volume
param (
    [string]$ContainerName = "qdrant-hybrid-rag",
    [int]$HttpPort = 6333,
    [int]$GrpcPort = 6334,
    [string]$VolumeName = "qdrant_storage"
)

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Starting Qdrant Docker Container for Hybrid RAG" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# Check if Docker daemon is accessible
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker daemon is not running or not accessible." -ForegroundColor Red
    Write-Host "Please start Docker Desktop on your machine and run this script again." -ForegroundColor Yellow
    exit 1
}

# Check if container already exists
$existing = docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}"
if ($existing -eq $ContainerName) {
    $status = docker inspect -f '{{.State.Running}}' $ContainerName
    if ($status -eq "true") {
        Write-Host "Qdrant container '$ContainerName' is already running." -ForegroundColor Green
    } else {
        Write-Host "Starting existing Qdrant container '$ContainerName'..." -ForegroundColor Yellow
        docker start $ContainerName
    }
} else {
    Write-Host "Creating and starting new Qdrant container '$ContainerName'..." -ForegroundColor Yellow
    docker run -d `
        --name $ContainerName `
        -p ${HttpPort}:6333 `
        -p ${GrpcPort}:6334 `
        -v ${VolumeName}:/qdrant/storage `
        --restart unless-stopped `
        qdrant/qdrant:v1.12.0
}

Write-Host "`nQdrant is ready!" -ForegroundColor Green
Write-Host "REST API:       http://localhost:${HttpPort}" -ForegroundColor White
Write-Host "Web Dashboard:  http://localhost:${HttpPort}/dashboard" -ForegroundColor White
Write-Host "gRPC Endpoint:  localhost:${GrpcPort}" -ForegroundColor White
Write-Host "====================================================" -ForegroundColor Cyan

