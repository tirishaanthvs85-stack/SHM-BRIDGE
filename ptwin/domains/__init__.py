"""Domains package for PTwin."""
from . import beam, pipeline, battery
from .base import PhysicsTwinProblem
from .beam import BeamDomain
from .pipeline import PipelineDomain
from .battery import BatteryDomain

DOMAINS = {"beam": BeamDomain, "pipeline": PipelineDomain, "battery": BatteryDomain}
