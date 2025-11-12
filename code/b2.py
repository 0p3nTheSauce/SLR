import subprocess

for i in range(8):
    print(f"{i}")
    try:
        subprocess.run(['python', 'benchmark.py', f'{i}'], check=True)
    except Exception as e:
        print(f'Model: {i} failed: ')
        print(e)