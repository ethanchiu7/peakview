#!/bin/bash
expect -c ' \
set login_pass 'password'
spawn ssh -L 16006:127.0.0.1:6006 luban@xx -p 8022
expect "*password*" {
send "$login_pass\r"
}
interact'
