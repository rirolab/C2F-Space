#!/usr/bin/env python3

"""Run file to execute the spatial localization pipeline. The parameters
of the pipeline can be set by passing arguments to this file.
"""

# Library imports
from src.core import spatial_reasoning

            
if __name__ == "__main__":
    """main function which accepts the arguments and sets appropriate variables
    """    
    
    # Run the pipeline
    spatial_reasoner = spatial_reasoning.SpatialReasoner()
    spatial_reasoner.run()