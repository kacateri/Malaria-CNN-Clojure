(defproject malariacnn "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.8.0"]
				 [org.clojure/tools.cli "0.3.5"]
				 [net.mikera/imagez "0.12.0"]
                 [thinktopic/experiment "0.9.22"]
                 [thinktopic/think.image "0.4.11"]
				 [thinktopic/think.datatype "0.3.17"]]
  :main ^:skip-aot malariacnn.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
