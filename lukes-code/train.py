import argparse





def main():
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser.add_argument('model', help='model to use')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-n', '--lines', type=int, default=10, help='Number of lines to process')
    
    args = parser.parse_args()
    
    print(f"Processing {args.input_file}")
    if args.output:
        print(f"Output will go to {args.output}")
    if args.verbose:
        print("Verbose mode enabled")
    print(f"Processing {args.lines} lines")

if __name__ == '__main__':
    main()