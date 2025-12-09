<h1 align='center'>mrid</h1>

mrid is a library for preprocessing of 3D images, particularly medical images.

It provide interfaces for many medical image processing tools such as [SimpleElastix](https://simpleelastix.github.io/), [HD-BET](https://github.com/MIC-DKFZ/HD-BET#Installation), [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/), [CTSeg](https://github.com/WCHN/CTseg). Note that those libraries are not bundled with mrid, I've included installation instructions in all notebooks.

### Installation

Either run

```
pip install mrid-python
```

or

```
pip install git+https://github.com/inikishev/mrid
```

### Basics

The images you pass to all functions in mrid can be path to a .nii.gz file, DICOM directory, sitk.Image, numpy array or torch tensor. All functions return results as `sitk.Image`. If you need a numpy array, you can use `mrid.tonumpy(sitk_image)`.

### Registering images with SimpleITK-SimpleElastix

[SimpleElastix](https://simpleelastix.github.io/) is a robust tool for image registration which works really well out-of-the-box. It works on both Windows and Linux.

IMAGE AND NOTEBOOK！！！！！

### Skullstripping MRI scans with HD-BET

[HD-BET](https://github.com/MIC-DKFZ/HD-BET) is a model that performs skullstripping of pre- and post-constrast T1, T2 and FALIR MRIs. It works on both Windows and Linux.

IMAGE AND NOTEBOOK！！！！！

### Skullstripping with SynthStrip
[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) is a skull-stripping tool that works with many different image types and modalities, including MRI, DWI, CT, PET, etc.

IMAGE AND NOTEBOOK！！！！！


### Skullstripping and segmentation of CT images with CTseg

TODO

### Example workflow - preprocessing MRIs to BraTS format

Many [BraTS](https://www.synapse.org/brats) datasets are provided as skullstripped images in SRI24 space. See this notebook for how to process raw scans to this format, and then process segmentations back to original format.

IMAGE AND NOTEBOOK！！！！！
