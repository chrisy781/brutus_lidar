
(cl:in-package :asdf)

(defsystem "pandar_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "PandarGps" :depends-on ("_package_PandarGps"))
    (:file "_package_PandarGps" :depends-on ("_package"))
    (:file "PandarPacket" :depends-on ("_package_PandarPacket"))
    (:file "_package_PandarPacket" :depends-on ("_package"))
    (:file "PandarScan" :depends-on ("_package_PandarScan"))
    (:file "_package_PandarScan" :depends-on ("_package"))
  ))