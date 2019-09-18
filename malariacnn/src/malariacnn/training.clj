(ns malariacnn.training
  (:require [clojure.java.io :as io])
  (:require [mikera.image.core :as imagez])
  (:require [cortex.experiment.util :as experiment-util])
  (:require [cortex.nn.layers :as layers])
  (:require [think.image.patch :as patch])
  (:require [cortex.experiment.classification :as classification])
)

(def dataset-folder "resources/")
(def training-folder (str dataset-folder "training"))
(def test-folder (str dataset-folder "testing"))
(def categories 
  (into [] (map #(.getName %) (.listFiles (io/file training-folder)))))
(def num-classes (count categories))
(def class-mapping
  {:class-name->index (zipmap categories (range))
   :index->class-name (zipmap (range) categories)})   
   
(def first-test-pic 
  (first (filter #(.isFile %) (file-seq (io/file training-folder)))))
(imagez/load-image first-test-pic)
(def image-size 
  (.getWidth (imagez/load-image first-test-pic)))

(def train-ds
 (-> training-folder
   (experiment-util/create-dataset-from-folder
     class-mapping :image-aug-fn (:image-aug-fn {}))
   (experiment-util/infinite-class-balanced-dataset)))

(def test-ds
 (-> test-folder
     (experiment-util/create-dataset-from-folder class-mapping)))

(defn initial-description
 [input-w input-h num-classes]
 [(layers/input input-w input-h 1 :id :data)
  (layers/convolutional 5 0 1 30)
  (layers/max-pooling 2 1 2)
  (layers/relu)
  (layers/convolutional 5 0 1 50)
  (layers/max-pooling 2 1 2)
  (layers/batch-normalization)
  (layers/linear 1000)
  (layers/relu :center-loss {:label-indexes {:stream :labels}
                             :label-inverse-counts {:stream :labels}
                             :labels {:stream :labels}
                             :alpha 0.9
                             :lambda 1e-4})
  (layers/linear num-classes)
  (layers/softmax :id :labels)])

(defn- observation->image
   "Creates a BufferedImage suitable for web display from the raw data
   that the net expects."
   [observation]
   (patch/patch->image observation image-size))

  
;(let [listener (classification/create-listener observation->image class-mapping {})]
;   (classification/perform-experiment
;   (initial-description image-size image-size num-classes)
;     train-ds
;     test-ds
;     listener))
