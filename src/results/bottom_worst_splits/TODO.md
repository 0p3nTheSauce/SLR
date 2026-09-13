The notebooks in this directory need updating in the following ways:
- Frame viewing: use the `FrameFetcher`/`FrameVisualiser` pattern where possible as shown in [class_viewer](../dataset_analysis/class_viewer.ipynb).
- Use consistent heading sections with markdown, so that the notebookes are nicer to read, and each relevant section can be closed with a drop down, merging code cells and markdown where necessary
- [100_worst](./100_worst.ipynb) loads a model manually an runs inference. A helper method called `infer` should be added to [visualise2](../../visualise2.py) which performs this functionality (loading a model and running a simple inferrence) in spirit of the `FrameFetcher` pattern for viewing frames. 
