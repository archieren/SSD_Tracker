;; Don't forget putting (add-hook 'hack-local-variables-hook (lambda () (editorconfig-apply))) to your .emacs.
((c++-mode  . ((flycheck-gcc-include-path . ("/home/archie/Projects/SSD_Tracker/include/"
                                             "/usr/include/opencv4/"
                                             "/usr/include/tensorflow/"
                                             "/usr/include/tensorflow/bazel-out/k8-opt/bin/"))
               ))

 (c-mode  . ((flycheck-gcc-include-path . ("/home/archie/Projects/SSD_Tracker/include/"
                                           "/usr/include/opencv4/"
                                           "/usr/include/tensorflow/"
                                           "/usr/include/tensorflow/bazel-out/k8-opt/bin/"))
             ))
 )

