"""
FastAPI server for Dripper HTML extraction service.

This module provides a REST API server that accepts HTML content and returns
extracted main HTML using the Dripper extraction engine.
"""

import argparse
import logging
import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dripper.api import Dripper

# -------------- Logging Configuration --------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger('dripper_server')

# -------------- Command Line Arguments --------------
parser = argparse.ArgumentParser(
    description='Dripper HTML extraction server'
)
parser.add_argument(
    '--model_path',
    type=str,
    default=None,
    help='Path to the LLM model (can also be set via DRIPPER_MODEL_PATH env var)'
)
parser.add_argument(
    '--state_machine',
    type=str,
    default=None,
    help='State machine version to use (can also be set via DRIPPER_STATE_MACHINE env var)'
)
parser.add_argument(
    '--port',
    type=int,
    default=7986,
    help='Port number to run the server on (can also be set via DRIPPER_PORT env var)'
)
args = parser.parse_args()

# -------------- Configuration --------------
# Configuration can be set via environment variables or command line arguments
# Environment variables take precedence
MODEL_PATH = os.getenv('DRIPPER_MODEL_PATH', args.model_path)
STATE_MACHINE = os.getenv('DRIPPER_STATE_MACHINE', args.state_machine)
PORT = int(os.getenv('DRIPPER_PORT', str(args.port)))

# -------------- Global Singleton --------------
# Initialize Dripper instance as a global singleton for reuse across requests
dripper = Dripper(
    config={
        'model_path': MODEL_PATH,
        'tp': 1,  # Tensor parallel size
        'state_machine': STATE_MACHINE,
        'use_fall_back': True,  # Use fallback mechanism on errors
        'raise_errors': False,  # Don't raise exceptions, return None instead
        'early_load': True,  # Load model early during initialization
    }
)

# -------------- FastAPI Application --------------
app = FastAPI(title='DripperBatch1', version='0.1.0')


class ExtractReq(BaseModel):
    """Request model for HTML extraction endpoint."""

    html: str = Field(..., description='Raw HTML string to extract main content from')


class ExtractResp(BaseModel):
    """Response model for HTML extraction endpoint."""

    main_html: str = Field(..., description='Extracted main HTML string')


@app.post('/extract', response_model=ExtractResp)
async def extract_main(req: ExtractReq) -> Dict[str, Any]:
    """
    Extract main HTML content from raw HTML.

    This is a synchronous single-item endpoint (batch=1) that directly calls
    Dripper.process to extract the main HTML content from the input HTML.

    Args:
        req: ExtractReq object containing the raw HTML string

    Returns:
        Dictionary containing the extracted main HTML string

    Raises:
        HTTPException: If extraction fails or no valid main HTML is found
    """
    try:
        # Process HTML using Dripper (batch size = 1)
        result_list = dripper.process(req.html)

        # Validate that we got a valid result
        if not result_list or result_list[0].main_html is None:
            raise RuntimeError('no valid main html')

        return {'main_html': result_list[0].main_html}
    except Exception as e:
        logger.exception('extract error')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
def health() -> Dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Dictionary with server status
    """
    return {'status': 'ok'}


if __name__ == '__main__':
    # Start the FastAPI server using uvicorn
    # The server runs on all network interfaces (0.0.0.0) and uses a single worker
    # process with uvloop for better async performance
    uvicorn.run(
        'server:app',
        host='0.0.0.0',  # Listen on all network interfaces
        port=PORT,
        workers=1,  # Single worker process
        loop='uvloop',  # Use uvloop for better async performance
    )
