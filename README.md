# TO DO:
- current status: reading files, calculating affines, saving metadata etc seems to be working fine for sitk, also when saving to nifti. however, saving using nibabel seems to be a bit wrong.
    - need to fix affine when saving with nibabel, currently it does not seem to get updated properly after resampling?
    - orientation also seems to be different from when saving with sitk, and this seems to cause some inconsistencies when loading with nibabel also.
- Figure out image orientations. It just needs to be consistent when using some package (probably sitk, even though I like nibabel better)

# notes
- 2D seems to be working

- [x] 3D does not yet work properly. I wanted to make something that worked with missing slices or uneven slice thickness (because converting using dicom2nifti does not work if you have inconsistent slice thickness). This went to shit though. Also, I have had issues with getting the affine correctly.

## notes on image orientation
- 