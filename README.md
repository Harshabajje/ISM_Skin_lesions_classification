# ISM

## TODO Phase 2
We get quite good performance now when using efficient net but there is still a lot to do. Some stuff I came up with:  
*italic* = probably not that important
* data augmentation:
	* [ ] what methods to use, how much augmentation
	* [x] how to crop/resize the images
	    * there are some different ways in the papers
	    * test resizing to different sizes before cropping
* *pre-processing:*
	* It might be helpful to do apply undistortion or color correction before the data augmentation (see ISIC papers)
	* -> might take a bit to long to implement 
* input strategy for validation (and test) data
	* [x] how many inputs per image, how to crop
		* we use the same augmentation as for training right now, this might not be optima
		* -> new version is implemented
	* [x] combine multiple models
		* some teams are doning this, we could check it out as well
		* -> tried it, didn't bring an improvement, could try with some other models
* pre-trained model:
	* [x] select best model(s) to use
	    * -> probably one (or multiple) of the efficient nets, they just work pretty good
	    * tested up to b4, could also try larger models
* hyper-parameter:
	* [x] *batch size*
	    * -> 32 if possible, lower as needed by GPU for larger models
	* [ ] *epochs of training*
	    * can use some callbacks to test this while training but works good enought 
	* [ ] learning rate
	* [ ] unfreezing
		* how many layer
		* -> more layers improved b3 to 79%
		* how many epochs
* unbalanced dataset:
	* [x] find best method to deal with the unbalanced dataset 
	    * class weights vs oversampling vs ...
	    * -> oversampling doesn't really work, class weights are good
* *external data*:
    * [ ] use some external datasets (7-point?) for unknown class
    * [ ] try to find a way to evaluate this before final upload
* unknown class
    * [x] implement handling of unknown category
        * do some research if confidence values (softmax output works for that)
        * try leaving one class out in training to simulate thi
        * -> works good enoug
    * [ ]  create prediction for testdataset
* **metadata**:
	* [ ] figure out how to include the metadata for task 2
