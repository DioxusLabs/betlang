(defpackage #:betlang-fixture
  (:use #:cl)
  (:export #:summarize))

(in-package #:betlang-fixture)

(defun normalize (value)
  (string-downcase (string-trim '(#\Space #\Tab #\Newline) value)))

(defun summarize (values)
  (loop with table = (make-hash-table :test #'equal)
        for value in values
        for key = (normalize value)
        do (incf (gethash key table 0))
        finally (return table)))

(format t "~A~%" (summarize '("Rust" "Rust" "Python")))
