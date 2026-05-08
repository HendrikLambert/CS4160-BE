# CS4160 Assignment 1

IPv8 client that mines an N-bit SHA-256 PoW over an email and this repo's URL, then submits it to the course server. 

To run, first install the requirements (`pip install -r requirements.txt`), then run the client (`python3 client.py`). Also supports GPU hashing on NVIDIA GPUs, this is based on a modified open source SHA256 kernel. Can be used with --gpu.
