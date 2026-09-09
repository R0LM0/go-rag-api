package application

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
)

// newID returns a cryptographically random 128-bit identifier encoded as 32
// lowercase hexadecimal characters. It deliberately uses only the standard
// library instead of a third-party UUID package.
func newID() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("reading random bytes for id: %w", err)
	}
	return hex.EncodeToString(buf), nil
}
