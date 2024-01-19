# README

Package to perform conversions between dicom and nifti file formats for medical images. It relies mainly on SimpleITK, but can also handle nibabel files to some extent. It also contains simple visualization functions. It doesn't do anything new, but is meant as a convenient package for some functions that I end up using often and want to treat in a consistent manner.

## dev notes
- notebooks/test-all.ipynb contains a demo of some test data, including both 2D and 3D images. It seems to work well.

## To do:
- implement pydicom version of all functions (started, but not finished)
- test to see that you get the same results in terms of nifti files regardless of using sitk or pydicom
- add tests of key functions

## release new version
`python3 -m build --wheel`