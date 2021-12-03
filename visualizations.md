## Visualizations

### Fine-tuned Parameters

Here we show some results of Yan's method and Cui's method. We tune the parameters for each dataset based on the pose graph of it. 

<details>
<summary>[Click to expand]</summary>

The gif on the upper left displays the images in each dataset. The gif on the upper right is the reconstructions from the `colmap` with default parameters. The bottom left and the bottom right gifs display the reconstructions after disambiguation with Yan's method and Cui's method, respectively.

<!--
One thing worth noticing is that the parameters used for different datasets are different: we kind of cheat by tuning the parameters based on the pose graphs.
-->

#### Books

<p float="left">
    <img src="visualizations/books_dat.gif" width="49%", alt="Books Dataset">
    <img src="visualizations/books_ori.gif" width="49%", alt="Books COLMAP">
</p>
<p float="left">
    <img src="visualizations/books_yan.gif" width="49%", alt="Books Yan's Method">
    <img src="visualizations/books_cui.gif" width="49%", alt="Books Cui's Method">
</p>

#### Cereal

<p float="left">
    <img src="visualizations/cereal_dat.gif" width="49%", alt="Cereal Dataset">
    <img src="visualizations/cereal_ori.gif" width="49%", alt="Cereal COLMAP">
</p>
<p float="left">
    <img src="visualizations/cereal_yan.gif" width="49%", alt="Cereal Yan's Method">
    <img src="visualizations/cereal_cui.gif" width="49%", alt="Cereal Cui's Method">
</p>

#### Cup

<p float="left">
    <img src="visualizations/cup_dat.gif" width="49%", alt="Cup Dataset">
    <img src="visualizations/cup_ori.gif" width="49%", alt="Cup COLMAP">
</p>
<p float="left">
    <img src="visualizations/cup_yan.gif" width="49%", alt="Cup Yan's Method">
    <img src="visualizations/cup_cui.gif" width="49%", alt="Cup Cui's Method">
</p>

#### Desk

<p float="left">
    <img src="visualizations/desk_dat.gif" width="49%", alt="Desk Dataset">
    <img src="visualizations/desk_ori.gif" width="49%", alt="Desk COLMAP">
</p>
<p float="left">
    <img src="visualizations/desk_yan.gif" width="49%", alt="Desk Yan's Method">
    <img src="visualizations/desk_cui.gif" width="49%", alt="Desk Cui's Method">
</p>

(the image on the most left is misregistered with `colmap`, while it is corrected with either one of the two methods)

#### Oats

<p float="left">
    <img src="visualizations/oats_dat.gif" width="49%", alt="Oats Dataset">
    <img src="visualizations/oats_ori.gif" width="49%", alt="Oats COLMAP">
</p>
<p float="left">
    <img src="visualizations/oats_yan.gif" width="49%", alt="Oats Yan's Method">
    <img src="visualizations/oats_cui.gif" width="49%", alt="Oats Cui's Method">
</p>

(both methods failed as the ground truth should be something like a sequence instead of two sequences in parallel)

#### Street

<p float="left">
    <img src="visualizations/street_dat.gif" width="49%", alt="Street Dataset">
    <img src="visualizations/street_ori.gif" width="49%", alt="Street COLMAP">
</p>
<p float="left">
    <img src="visualizations/street_yan.gif" width="49%", alt="Street Yan's Method">
    <img src="visualizations/street_cui.gif" width="49%", alt="Street Cui's Method">
</p>

#### Temple of Heaven

<p float="left">
    <img src="visualizations/ToH_dat.gif" width="49%", alt="ToH Dataset">
    <img src="visualizations/ToH_ori.gif" width="49%", alt="ToH COLMAP">
</p>
<p float="left">
    <img src="visualizations/ToH_yan.gif" width="49%", alt="ToH Yan's Method">
    <img src="visualizations/ToH_cui.gif" width="49%", alt="ToH Cui's Method">
</p>

#### Alexander Nevsky Cathedral

<p float="left">
    <img src="visualizations/alex_dat.gif" width="49%", alt="Alexander Nevsky Cathedral Dataset">
    <img src="visualizations/alex_ori.gif" width="49%", alt="Alexander Nevsky Cathedral COLMAP">
</p>
<p float="left">
    <img src="visualizations/alex_yan.gif" width="49%", alt="Alexander Nevsky Cathedral Yan's Method">
    <img src="visualizations/alex_cui.gif" width="49%", alt="Alexander Nevsky Cathedral Cui's Method">
</p>

</details>

### Same Parameters

To investigate to what extent a set of parameters would be applicable for all datasets, we apply the parameters tuned for the Alexander Nevsky Cathedral dataset on other Internet collections of images.

<details>
<summary>[Click to expand]</summary>

#### Arc de Triomphe

<p float="left">
    <img src="visualizations/arc_de_triomphe_1.jpg" width="49%">
    <img src="visualizations/arc_de_triomphe_2.jpg" width="49%">
</p>
<p float="left">
    <img src="visualizations/arc_de_triomphe.png" width="100%">
</p>

### Berliner Dom

<p float="left">
    <img src="visualizations/berliner_dom_1.jpg" width="49%">
    <img src="visualizations/berliner_dom_2.jpg" width="49%">
</p>
<p float="left">
    <img src="visualizations/berliner_dom.png" width="100%">
</p>

#### Big Ben

<p float="left">
    <img src="visualizations/big_ben_1.jpg" width="49%">
    <img src="visualizations/big_ben_2.jpg" width="49%">
</p>
<p float="left">
    <img src="visualizations/big_ben.png" width="100%">
</p>

#### Brandenburg Gate

<p float="left">
    <img src="visualizations/brandenburg_gate_1.jpg" width="49%">
    <img src="visualizations/brandenburg_gate_2.jpg" width="49%">
</p>
<p float="left">
    <img src="visualizations/brandenburg_gate.png" width="100%">
</p>

(With a proper choice of the threshold, we can disambiguate the model into several parts.)

#### Church of Savior on the Spilled Blood

<p float="left">
    <img src="visualizations/church_on_spilled_blood_1.jpg" width="49%">
    <img src="visualizations/church_on_spilled_blood_2.jpg" width="49%">
</p>
<p float="left">
    <img src="visualizations/church_on_spilled_blood.png" width="100%">
</p>

(With a proper choice of the threshold, we can disambiguate the model into several parts.)

#### Radcliffe Camera

<p float="left">
    <img src="visualizations/radcliffe_camera_1.jpg" width="49%">
    <img src="visualizations/radcliffe_camera_2.jpg" width="49%">
</p>
<p float="left">
    <img src="visualizations/radcliffe_camera.png" width="100%">
</p>

(The correct reconstruction is split into two parts due to the lack of transitional camera views)

</details>

### Kataria's Method

Here we would like to also display the results from the paper [Improving Structure from Motion with Reliable Resectioning](https://rajbirkataria.com/assets/ImprovingStructurefromMotionwithReliableResectioning.pdf) by Rajbir Kataria, Joseph DeGol, Derek Hoiem. For more details, please refer to the [repository](https://github.com/rajkataria/ReliableResectioning) provided by the authors. We refer to this method as Kataria's method hereafter.

<details>
<summary>[Click to expand]</summary>

Based on the observation that longer tracks are more likely to contain wrong matches, the authors propose to use a track-length-adjusted number of matches as the criterion for the next view selection. More importantly, the initial pose of the image to be registered will rely only on 3D points from reliable images instead of all triangulated points. This is important as our experiments show that a correct registration order does not necessarily lead to a correct reconstruction. This method only contains two parameters to be set: the track length discount factor &lambda; and the reliable image threshold &tau;. More significantly, the same set of parameters could work on many different scenes, greatly reducing the burden of tuning parameters for the above mentioned two methods.

For a fair comparison, we investigate the changed files in the original repository and integrate them with the current version of colmap with small modifications. We run the `exhaustive_matcher` instead of `vocab_tree_matcher` as done in previous methods. Since the parameters provided by the author are tuned for OpenSfm, we also tried to tune the parameters (&lambda; changed from 0.5 to 0.3, &tau; changed from 2.0 to 1.3) for colmap on the cup and the oats dataset. The results are shown below:

#### Cup

<p float="left">
    <img src="visualizations/cup_rr.png" width="49%">
    <img src="visualizations/cup_rr_tune.png" width="49%">
</p>
(The reconstruction on the left is with the parameters provided by the authors, while the one on the right is with the parameters tuned by us)

#### Oats

<p float="left">
    <img src="visualizations/oats_rr.png" width="49%">
    <img src="visualizations/oats_rr_tune.png" width="49%">
</p>
(The reconstruction on the left is with the parameters provided by the authors, while the one on the right is with the parameters tuned by us. Note that we did not find a set of suitable parameters for Yan's or Cui's method to disambiguate this scene)

#### Results on Large Scale Datasets

However, when we use these two sets of parameters on the large scale Internet datasets provided by Heinly et al, both sets of the parameters give us similar reconstructions and they are somewhat inferior to what we can get from Yan's or Cui's method:

#### Alexander Nevsky Cathedral

<p float="left">
    <img src="visualizations/alex_rr_tune.png" width="49%">
    <img src="visualizations/alex_yan.png" width="49%">
</p>

(In the left reconstruction, some of the misregistered cameras should be placed in the blue circle to create a correct reconstruction like the one on the right)

#### Big Ben

<p float="left">
    <img src="visualizations/big_ben_rr_tune.png" width="49%">
    <img src="visualizations/big_ben_cui.png" width="49%">
</p>

(Note the suspicious wall in the blue circle in the left reconstruction, which should be an empty street as in the right reconstruction)

#### Radcliffe Camera

<p float="left">
    <img src="visualizations/radcliffe_camera_rr_tune_yan.png" width="100%">
</p>

(This set of parameters for Kataria's method cannot distinguish the two sides of Radcliffe Camera, while Yan's method and Cui's method work)

### Reproduction

For the reproduction of the above results for Kataria's method, we put the changed/added files in the [reliable_resectioning](./reliable_resectioning/src) folder. You can merge all the files in this directory with colmap's source code and then compile it. We also provide a [bash script example](./scripts/reliable_resectioning_exhaustive_colmap.sh) for generating sparse reconstruction with the newly compiled colmap.

</details>