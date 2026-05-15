select users.id, users.email
from users
where users.active = true
order by users.id;
