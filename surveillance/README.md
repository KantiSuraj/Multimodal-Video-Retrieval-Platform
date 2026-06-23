uvicorn services.embedding.main:app     --host 0.0.0.0     --port 8003     --reload
uvicorn services.detection.main:app     --host 0.0.0.0     --port 8002     --reload
uvicorn services.preprocessing.main:app     --host 0.0.0.0     --port 8001     --reload
uvicorn services.ingestion.main:app     --host 0.0.0.0     --port 8000     --reload



curl localhost:8001/health

export PYTHONPATH=$PWD
MINIO_ROOT_USER=minioadmin \
MINIO_ROOT_PASSWORD=minioadmin \
minio server ~/minio-data --console-address ":9001"
sudo systemctl start postgresql
sudo systemctl start rabbitmq-server
sudo systemctl start redis-server

verify postgressql
psql -h localhost -U postgres -d surveillance
postgres
SELECT
    id,
    status,
    error_message
FROM video_records
ORDER BY created_at DESC
LIMIT 5;
\dT+ videostatus

alembic -c infra/migrations/alembic.ini upgrade head

verify detection
SELECT *
FROM detection_results
LIMIT 10;

curl -X POST \
http://localhost:8000/api/v1/videos \
-F "file=@/path/to/real_video.mp4"


sudo -u postgres psql
pg_isready -U postgres
redis-cli ping
sudo rabbitmqctl status


SELECT id, sha256_hash, original_filename
FROM video_records;

DELETE FROM detection_results
WHERE video_id = '7a98ef02-965e-4058-862d-d5a4dfbcb5a7';

DELETE FROM embedding_records
WHERE video_id = '7a98ef02-965e-4058-862d-d5a4dfbcb5a7';

DELETE FROM video_records
WHERE id = '7a98ef02-965e-4058-862d-d5a4dfbcb5a7';

curl -X POST \
http://localhost:8000/api/v1/videos \
-F "file=@/home/kanti_suraj/Desktop/VIRAT_S_010204_05_000856_000890.mp4"
7a98ef02-965e-4058-862d-d5a4dfbcb5a7
