#!/usr/bin/env python3
import requests
import time
import json

def test_motion_service():
    """Test the motion generation service"""
    
    base_url = "http://localhost:8010"
    
    # Test request
    motion_request = {
        "text": "A person is walking confidently with a swagger",
        "duration": 5.0
    }
    
    print("🚀 Submitting motion generation request...")
    print(f"Text: {motion_request['text']}")
    print(f"Duration: {motion_request['duration']} seconds")
    
    # Submit request
    response = requests.post(f"{base_url}/upload", json=motion_request)
    
    if response.status_code != 200:
        print(f"❌ Error submitting request: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    task_id = result["task_id"]
    print(f"✅ Task submitted successfully! Task ID: {task_id}")
    
    # Poll for status
    print("\n📊 Monitoring task status...")
    while True:
        response = requests.get(f"{base_url}/status/{task_id}")
        
        if response.status_code != 200:
            print(f"❌ Error checking status: {response.status_code}")
            break
        
        status = response.json()
        print(f"Status: {status['status']} | Progress: {status.get('progress', 0)*100:.1f}% | Message: {status.get('message', 'N/A')}")
        
        if status["status"] == "completed":
            print("🎉 Task completed successfully!")
            
            # Get download URL
            response = requests.get(f"{base_url}/download/{task_id}")
            if response.status_code == 200:
                download_info = response.json()
                print(f"📥 Download URL: {download_info['download_url']}")
                
                # Optional: Download the file
                print("⬇️ Downloading BVH file...")
                bvh_response = requests.get(download_info['download_url'])
                if bvh_response.status_code == 200:
                    with open(f"generated_motion_{task_id}.bvh", "w") as f:
                        f.write(bvh_response.text)
                    print(f"✅ BVH file saved as: generated_motion_{task_id}.bvh")
                else:
                    print(f"❌ Error downloading file: {bvh_response.status_code}")
            else:
                print(f"❌ Error getting download URL: {response.status_code}")
            break
            
        elif status["status"] == "failed":
            print(f"❌ Task failed: {status.get('message', 'Unknown error')}")
            break
        
        time.sleep(2)

def test_health():
    """Test the health endpoint"""
    response = requests.get("http://localhost:8010/health")
    if response.status_code == 200:
        print("✅ Service is healthy")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

if __name__ == "__main__":
    print("🧪 Testing Motion Generation Service")
    print("=" * 50)
    
    # Test health first
    test_health()
    print()
    
    # Test motion generation
    test_motion_service()