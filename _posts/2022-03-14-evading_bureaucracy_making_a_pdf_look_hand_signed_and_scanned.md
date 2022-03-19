---
layout: post
title: "Evading Bureaucracy: Making a PDF Look Hand-Signed and Scanned"
description: "How I added a signature to a PDF document and made it look like a scanned copy."
categories: ["blog"]
tags: miscellaneous
---

__Update (03/18/2022)__: _So apparently my landlord saw the first version of this post, which prompted him "retract" the initial lease and angrily ask for a wet signature (I temporarily [removed this post](https://github.com/fzliu/fzliu.github.io/commit/99d18fe826f616e3f16ad5a2c3fc1504eaa18d6e) due to this). Instead of obliging, I used the same [Towhee pipeline](https://towhee.io/fzliu/sign-and-scan) as described below, but randomized the rotation of each page and slightly modified the way noise was added._

_He accepted the resulting document (just as he did the first, before seeing this post). I believe he found my website because I sent him my LinkedIn profile prior to signing the lease._

----

#### Moving to a new place

I recently moved into a new place - with this, I had the immense pleasure of signing a new lease. Initially, I scribbled a digital signature on the lease document using my trackpad and sent it my soon-to-be landlord. It looked fairly blocky and like it was sketched out using a computer mouse, but it could still easily be matched to my "actual" pen-and-paper signature. A day later, I had this conversation with him:

- __landlord__: _hi Frank, please send me a signed and scanned copy of your lease agreement_
- __me__: _hey REDACTED, I already signed it and sent it to your email_
- __landlord__: _the copy that you sent me does not look signed with a pen_
- __me__: _correct, I signed it digitally_
- __landlord__: _this type of signature is unacceptable for me_
- __landlord__: _please print it and sign it with a pen, then scan it send it back to me_
- __me__: _I unfortunately do not have access to a printer or scanner_
- __landlord__: _there are plenty of copy stores around San Francisco_

And with that, my software engineering spidey sense kicked into high gear...

#### Down with the bureaucrats

The honest truth is that I did have access to a printer and scanner (in the [Zilliz](https://zilliz.com) office), but did not want to go all the way there just to print, sign, scan, and send the final document. I also felt uncomfortable using company resources for this purpose. So I instead turned to code.

__Uploading a hand-written signature__

The first step was to create a hand-written signature that looked like it was scanned and uploaded. For this, I first put pen to paper and took a close-up picture of my "signature". I then downsized this image using (Imagemagick)[https://imagemagick.org]. This step is important, as it aligns the size of the signature with the size of the eventual scanned document:

```shell
% mogrify -resize 720x720> -path signature.jpg
```

<div align="center">
  <img src="/img/fake-signature.jpg">
</div>
<p style="text-align:center"><sub>Not my real signature...</sub></p>

The next step here is to crop out the region with the signature. The first instinct of a machine learning engineer might be to apply a backbone-based detection model such as [SSD](https://arxiv.org/abs/1512.02325) or [YOLO](https://arxiv.org/abs/2004.10934), but for a constrained problem such as this one, it's much more practical to go for a traditional edge or keypoint detector rather than a heavier machine learned model.

I fell back to the beloved [SIFT algorithm](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) for keypoint detection<sup>1</sup>. The trusty [OpenCV](https://opencv.org/) library has a great implementation of this. Since I wasn't concerned with the keypoint quantity, I set a fairly high threshold value for low-contrast keypoint removal - this would throw away keypoints located in "whitespace" while preserving most of the keypoints located directly on the signature itself. The original SIFT paper used a threshold of `0.03`; I used a threshold of `0.1`, but some further experimentation may be needed here to determine an optimal value. In Python:

```python
>>> import cv2
>>> sift = cv2.SIFT_create(contrastThreshold=0.1)
>>> img = cv2.imread('signature.jpg')
>>> img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
>>> kps = sift.detect(img_gray)
```

Here's a visualization of the detected keypoints:

```python
>>> img_kps = cv2.drawKeypoints(img, kps, None)
>>> cv2.imshow('Keypoints', img_kps)
>>> cv2.waitKey(0)
>>> cv2.destroyAllWindows()
```

<div align="center">
  <img src="/img/fake-signature-with-keypoints.jpg">
</div>
<p style="text-align:center"><sub>Signature with SIFT keypoints overlaid.</sub></p>

Since an occasional outlier keypoint may still be detected, an optional step here would be to use DBSCAN or mean-shift clustering to group and extract keypoints near the center of the image (where the signature should be located). This would theoretically help improve signature detection for noisy images (I'll leave this for future work).

Now it was simply a matter of finding the leftmost, rightmost, uppermost, and lowermost keypoints and recording their respective locations within the image. I also added 10 pixels of padding to ensure the entire signature was captured.

```python
>>> import numpy as np
>>> kp_pts = np.array([kp.pt for kp in kps])
>>> x_min, x_max = np.min(kp_pts[:,0]), np.max(kp_pts[:,0])
>>> x_min, x_max = max(x_min - 10, 0), min(x_max + 10, img.shape[1])
>>> y_min, y_max = np.min(kp_pts[:,1]), np.max(kp_pts[:,1])
>>> y_min, y_max = max(y_min - 10, 0), min(y_max + 10, img.shape[0])
>>> print(x_min, x_max)
>>> print(y_min, y_max)
```

    226.4580535888672 504.46112060546875
    230.42076110839844 285.8047790527344

Now, cropping the signature can be done simply by slicing the input image using the computed bounding box:

```python
>>> img_crop = img[int(y_min):int(y_max),int(x_min):int(x_max)]
>>> img_gray_crop = img_gray[int(y_min):int(y_max),int(x_min):int(x_max)]
```

<sup>1</sup><sub>Everybody who knows me well knows that I have a lot of love for SIFT.</sub>

__"Signing" the document__

With a cropped version of my signature ready, the next step was to turn it into a black-and-white version. Recalling the concept of [image histograms](https://en.wikipedia.org/wiki/Image_histogram), I simply set a threshold value to determine which pixels should be black and which should be white; pixels less than this value would be assigned a value of `0` while pixels greater than this value would be assigned `255` (assuming 8-bit grayscale). To get rid of the "blocky" feel of black-and-white images, I blurred the resulting image using a Gaussian kernel.

```python
>>> img_bw_crop = np.copy(img_gray_crop)
>>> img_bw_crop[np.where(img_gray_crop < 80)] = 0
>>> img_bw_crop[np.where(img_gray_crop >= 80)] = 255
>>> img_bw_crop = cv2.GaussianBlur(img_bw_crop, (3, 3), 0.5)
>>> cv2.imwrite('signature-cropped.jpg', img_bw_crop)
```

<div align="center">
  <img src="/img/fake-signature-cropped.jpg">
</div>
<p style="text-align:center"><sub>Cropped signature.</sub></p>

Armed with this black-and-white signature, I now needed to copy the signature into the lease document. To do this, I used [Imagemagick](https://imagemagick.org), a handy image editing tool that can be called from the command line. I used the following subcommands to overlay my signature image into the original PDF lease:

- `density`, to control the DPI of the resulting document
- `gravity`, to determine the "anchor point" of the cropped signature
- `geometry`, to position the signature at the correct location

```shell
% composite -density 150 -gravity NorthWest -geometry +200+520 signature-cropped.jpg lease.pdf signed-lease.pdf
```

Positioning the signature (`-geometry +250+520`) can take a bit of trial and error, but once that was done I got something like this:

<div align="center">
  <img src="/img/fake-signed-lease.jpg">
</div>
<p style="text-align:center"><sub>Signed lease (the black boxes are redacted portions).</sub></p>

__Simulate a document scan__

Now was the tricky part - making the PDF look like a scanned copy. For this, I again turned to [Imagemagick](https://imagemagick.org), this time using a combination of the following options:

- `density`, again to control the DPI of the resulting document
- `rotate`, to artificially give each page a bit of random rotation
- `attenuate`, to control the amount of artificial noise when using `-noise`
- `noise`, to add some random white noise to the document
- `colorspace`, to emulate a standard scanner's black-and-white setting
- `blur`, to artificially add some blur to the output image

```shell
% convert -density 150 signed-lease.pdf -rotate -0.66 -attenuate 0.2 +noise Multiplicative -colorspace Gray -blur 3x0.5 signed-lease-scanned.pdf
```

<div align="center">
  <img src="/img/fake-signed-lease-scanned.jpg">
</div>
<p style="text-align:center"><sub>Done!</sub></p>

#### Running your own `pdf2scan` pipeline

I compiled this code into a [Towhee pipeline](https://towhee.io/fzliu/sign-and-scan). Do note that you'll need [Imagemagick](https://imagemagick.org) pre-installed on your computer, since it's unfortunately not pip-installable. If you're on Debian Linux (this includes Ubuntu):

```shell
$ apt install imagemagick
$ pip3 install towhee
```

If you're on MacOS:

```shell
% brew install imagemagick
% pip3 install towhee
```

Then, from within a Python terminal:

```python
>>> from towhee import pipeline
>>> p = pipeline('fzliu/sign-and-scan')
>>> p('/path/to/lease.pdf', '/path/to/output.pdf', '/path/to/signature/file.jpg', x_offset, y_offset)
```

In the example above, `x_offset` and `y_offset` would correspond to `250` and `520`, respectively. You can also input `None` as the third parameter to forgo signing the document, i.e. the output will be only an emulated scan.

#### Future work

I hope this tool proves to be useful for everybody out there wading through the endless sea of bureaucracy. Sometime in the near future I hope to improve on this pipeline by adding OCR along with a word vectorization model to automatically determine the correct place to put the signature.

As always, feel free to leave comments below. You can also connect with me via [Twitter](https://twitter.com/frankzliu) and [LinkedIn](https://linkedin.com/in/fzliu).
