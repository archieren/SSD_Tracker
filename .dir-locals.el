;; Don't forget putting (add-hook 'hack-local-variables-hook (lambda () (editorconfig-apply))) to your .emacs.
((c++-mode  . ((flycheck-gcc-include-path . ("/home/archie/Projects/SSD_Tracker/include/"
                                             "/usr/include/opencv4/"))
               ))

 (c-mode  . ((flycheck-gcc-include-path . ("/home/archie/Projects/SSD_Tracker/include/"
                                           "/usr/include/opencv4/"))
             ))
 )

