# HR-F02: guarded predecessor capture order

The integration audit found a runtime construction defect: it attempted to snapshot the predecessor's guarded knowledge after registering new reading parameters. The existing applicability guard correctly rejected the changed owner identity. Capture now occurs before attaching reading parameters, followed by the established `learner.rebind` validation on the complete new owner. The guard remains unchanged. No teaching, model selection or final evaluation is repeated; only the failed integration audit is resumed as a new owned verification job.
