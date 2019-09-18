(ns malariacnn.dataprep
  (:require [mikera.image.core :as imagez]
            [mikera.image.filters :as filters]
            [clojure.string :as string]
            [clojure.java.io :as io]))

(defn preprocess-image
  [output-dir image-size [idx [file label]]]
  (let [img-path (str output-dir "/" label "/" idx ".png" )]
    (when-not (.exists (io/file img-path))
      (println "> " img-path)
      (io/make-parents img-path)
      (-> (imagez/load-image file)
          ((filters/grayscale))
          (imagez/resize image-size image-size)
          (imagez/save img-path)))))

(defn- gather-files [path]
  (->> (io/file path)
       (file-seq)
       (filter #(.isFile %))))

(defn- produce-indexed-data-label-seq
 [files]
 (->> (map (fn [file] [file (-> (.getName file) (string/split #"\.") first)]) files)
      (map-indexed vector)))

(defn build-image-data
  [original-data-dir training-dir testing-dir target-image-size]
  (let [files (gather-files original-data-dir)
        pfiles (partition (int (/ (count files) 2)) (shuffle files))
        training-observation-label-seq 
          (produce-indexed-data-label-seq (first pfiles))
        testing-observation-label-seq 
          (produce-indexed-data-label-seq (last pfiles))
        train-fn (partial preprocess-image training-dir target-image-size)
        test-fn (partial preprocess-image  testing-dir target-image-size)]
  (dorun (pmap train-fn training-observation-label-seq))
  (dorun (pmap test-fn testing-observation-label-seq))))


 ; (build-image-data
 ;   "resources/original"
 ;   "resources/training"
 ;   "resources/testing"
 ;   52
 ; )

