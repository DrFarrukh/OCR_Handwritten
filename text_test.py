import moondream as md
from PIL import Image

# Option A: Moondream Cloud
model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI1YTk4NjkzNi02ZWRmLTQwYWUtODQwYS00NzU3YzBmODFmYjQiLCJvcmdfaWQiOiJLTWoxVHhTMGF1ZVN4MkVnekdZMjZnZVNYTFJWUnhyZCIsImlhdCI6MTc0NTgzMDk1NiwidmVyIjoxfQ.sT7l7x4WGAfNfy7IGyar-JaAG8JKCHYaEEyhgFCQvf4")

# Option B: Local Server
# model = md.vl(endpoint="http://localhost:2020/v1")

image = Image.open("111.jpg")

# Ask a question
result = model.query(image, "Convert this handwritten text to computer typography")
answer = result["answer"]
# request_id = result["request_id"]
print(f"Answer: {answer}")
# print(f"Request ID: {request_id}")

