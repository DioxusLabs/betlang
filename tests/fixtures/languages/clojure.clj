(ns betlang.fixture
  (:require [clojure.string :as str]))

(defn normalize [value]
  (-> value
      str/trim
      str/lower-case
      (str/replace #"\s+" "-")))

(defn summarize [rows]
  (reduce (fn [acc row]
            (update acc (normalize (:language row)) (fnil inc 0)))
          {}
          rows))

(println (summarize [{:language "Rust"} {:language "Rust"} {:language "Python"}]))
