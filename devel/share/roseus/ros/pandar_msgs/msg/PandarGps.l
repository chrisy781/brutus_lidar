;; Auto-generated. Do not edit!


(when (boundp 'pandar_msgs::PandarGps)
  (if (not (find-package "PANDAR_MSGS"))
    (make-package "PANDAR_MSGS"))
  (shadow 'PandarGps (find-package "PANDAR_MSGS")))
(unless (find-package "PANDAR_MSGS::PANDARGPS")
  (make-package "PANDAR_MSGS::PANDARGPS"))

(in-package "ROS")
;;//! \htmlinclude PandarGps.msg.html


(defclass pandar_msgs::PandarGps
  :super ros::object
  :slots (_stamp _used _year _month _day _hour _minute _second _fineTime _flag ))

(defmethod pandar_msgs::PandarGps
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:used __used) 0)
    ((:year __year) 0)
    ((:month __month) 0)
    ((:day __day) 0)
    ((:hour __hour) 0)
    ((:minute __minute) 0)
    ((:second __second) 0)
    ((:fineTime __fineTime) 0)
    ((:flag __flag) 0)
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _used (round __used))
   (setq _year (round __year))
   (setq _month (round __month))
   (setq _day (round __day))
   (setq _hour (round __hour))
   (setq _minute (round __minute))
   (setq _second (round __second))
   (setq _fineTime (round __fineTime))
   (setq _flag (round __flag))
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:used
   (&optional __used)
   (if __used (setq _used __used)) _used)
  (:year
   (&optional __year)
   (if __year (setq _year __year)) _year)
  (:month
   (&optional __month)
   (if __month (setq _month __month)) _month)
  (:day
   (&optional __day)
   (if __day (setq _day __day)) _day)
  (:hour
   (&optional __hour)
   (if __hour (setq _hour __hour)) _hour)
  (:minute
   (&optional __minute)
   (if __minute (setq _minute __minute)) _minute)
  (:second
   (&optional __second)
   (if __second (setq _second __second)) _second)
  (:fineTime
   (&optional __fineTime)
   (if __fineTime (setq _fineTime __fineTime)) _fineTime)
  (:flag
   (&optional __flag)
   (if __flag (setq _flag __flag)) _flag)
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; uint32 _used
    4
    ;; uint16 _year
    2
    ;; uint16 _month
    2
    ;; uint16 _day
    2
    ;; uint16 _hour
    2
    ;; uint16 _minute
    2
    ;; uint16 _second
    2
    ;; uint32 _fineTime
    4
    ;; uint32 _flag
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;; uint32 _used
       (write-long _used s)
     ;; uint16 _year
       (write-word _year s)
     ;; uint16 _month
       (write-word _month s)
     ;; uint16 _day
       (write-word _day s)
     ;; uint16 _hour
       (write-word _hour s)
     ;; uint16 _minute
       (write-word _minute s)
     ;; uint16 _second
       (write-word _second s)
     ;; uint32 _fineTime
       (write-long _fineTime s)
     ;; uint32 _flag
       (write-long _flag s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _used
     (setq _used (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint16 _year
     (setq _year (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _month
     (setq _month (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _day
     (setq _day (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _hour
     (setq _hour (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _minute
     (setq _minute (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint16 _second
     (setq _second (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; uint32 _fineTime
     (setq _fineTime (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _flag
     (setq _flag (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get pandar_msgs::PandarGps :md5sum-) "a075038e2dd27f000f34f827c3e40eec")
(setf (get pandar_msgs::PandarGps :datatype-) "pandar_msgs/PandarGps")
(setf (get pandar_msgs::PandarGps :definition-)
      "time stamp
uint32 used
uint16 year
uint16 month
uint16 day
uint16 hour
uint16 minute
uint16 second
uint32 fineTime
uint32 flag
")



(provide :pandar_msgs/PandarGps "a075038e2dd27f000f34f827c3e40eec")


