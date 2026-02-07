import subprocess
import sys
import os
import time
import threading
from pathlib import Path


def start_service(service_name, directory, command, port):
    """Start a service in a separate thread"""
    print(f"Starting {service_name} on port {port}...")
    
    try:
        # Change to the service directory
        os.chdir(directory)
        
        # Start the process
        if sys.platform.startswith('win'):
            process = subprocess.Popen(command, shell=True)
        else:
            process = subprocess.Popen(command, shell=True)
        
        # Change back to the root directory
        os.chdir('..')
        
        # Wait for the process to finish
        process.wait()
        
    except KeyboardInterrupt:
        print(f"\nStopping {service_name}...")
        process.terminate()
        process.wait()


def main():
    print("Starting Task Management System...")
    
    # Define services to start
    services = [
        {
            'name': 'MCP Server',
            'dir': 'mcp_server',
            'cmd': 'python main.py',
            'port': 8001
        },
        {
            'name': 'Backend Server', 
            'dir': 'backend',
            'cmd': 'python main.py',
            'port': 8000
        },
        {
            'name': 'Frontend',
            'dir': 'frontend',
            'cmd': 'npm run dev' if os.path.exists('frontend/package.json') else 'npx next dev',
            'port': 3000
        }
    ]
    
    threads = []
    
    # Start each service in a separate thread
    for service in services:
        if not Path(service['dir']).exists():
            print(f"Warning: Directory {service['dir']} does not exist, skipping {service['name']}")
            continue
            
        thread = threading.Thread(
            target=start_service,
            args=(service['name'], service['dir'], service['cmd'], service['port'])
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
        
        # Give time for each service to start before starting the next
        time.sleep(3)
    
    print("\nAll services started!")
    print("- MCP Server: http://localhost:8001")
    print("- Backend: http://localhost:8000")
    print("- Frontend: http://localhost:3000")
    print("\nPress Ctrl+C to stop all services.")
    
    try:
        # Keep the main thread alive
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        sys.exit(0)


if __name__ == "__main__":
    main()