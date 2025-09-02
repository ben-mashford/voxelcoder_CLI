#!/usr/bin/env python3
"""
Batch Alignment CLI Tool
Main entry point for the command-line batch alignment application.
"""

import argparse
import sys
import logging
from pathlib import Path

from config import load_config, validate_config
from utils import setup_logging, check_dependencies
from alignment import BatchAlignmentProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Batch Alignment CLI Tool - Align flow cytometry batches using autoencoders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --config config.yaml
  python cli.py --config config.yaml --validate
  python cli.py --config config.yaml --dry-run --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        type=Path,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate configuration and files without running alignment'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually running'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip pre-flight validation (not recommended)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if not args.config.exists():
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
            
        config = load_config(args.config)
        
        # Setup logging
        log_level = "DEBUG" if args.verbose else config.get('logging', {}).get('level', 'INFO')
        log_file = config.get('logging', {}).get('log_file')
        setup_logging(log_level, log_file)
        
        logger = logging.getLogger(__name__)
        logger.info(f"🚀 Starting Batch Alignment CLI")
        logger.info(f"📁 Configuration file: {args.config}")
        logger.info(f"🧪 Experiment: {config['experiment']['name']}")
        
        # Check dependencies
        check_dependencies()
        
        # Validate configuration
        if not args.skip_validation:
            logger.info("🔍 Validating configuration...")
            validation_errors = validate_config(config)
            
            if validation_errors:
                logger.error("❌ Configuration validation failed:")
                for error in validation_errors:
                    logger.error(f"   • {error}")
                sys.exit(1)
            
            logger.info("✅ Configuration validation passed")
        
        if args.validate:
            logger.info("✅ Validation complete. Exiting as requested.")
            return
        
        # Initialize processor
        processor = BatchAlignmentProcessor(config)
        
        if args.dry_run:
            logger.info("🔍 Dry run mode - showing what would be processed:")
            processor.show_processing_plan()
            return
        
        # Run the alignment process
        logger.info("🎯 Starting batch alignment process...")
        success = processor.run()
        
        if success:
            logger.info("🎉 Batch alignment completed successfully!")
            logger.info(f"📊 Results saved to: {config['experiment']['output_dir']}")
        else:
            logger.error("❌ Batch alignment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()