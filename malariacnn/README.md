# malariacnn

Convolutional neural network for malaria detection based od blood cell image in Clojure

## Installation

Download dataset from https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria.

## Usage

DATA

Download dataset. Rename images to class_name.image_number (for example parazitised.100.png or uninfected.55.png).
Add folder original to resources. Put all images (from both classes) directly into this folder.
Run dataprep.clj

TRAINING
Run training.clj

TEST
Run example.clj to test network performance with new image

    $ java -jar malariacnn-0.1.0-standalone.jar [args]


## License

Copyright © 2019 FIXME

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.

This Source Code may also be made available under the following Secondary
Licenses when the conditions for such availability set forth in the Eclipse
Public License, v. 2.0 are satisfied: GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or (at your
option) any later version, with the GNU Classpath Exception which is available
at https://www.gnu.org/software/classpath/license.html.
